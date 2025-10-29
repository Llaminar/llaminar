/**
 * @file IQ1_MTensor.cpp
 * @brief IQ1_M quantized tensor implementation (1-bit medium IQ)
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
    IQ1_MTensor::IQ1_MTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
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

    std::unique_ptr<ITensorGemm> IQ1_MTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ1_MTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(raw_data_.data());
        const IQ1_MBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ1_MTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ1_MTensor::decodeBlock(const IQ1_MBlock &block, float *output)
    {
        // IQ1_M: 256 elements per super-block
        // Extract global FP16 scale from packed scales array
        const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
        uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                             ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = fp16_to_fp32(scale_u16);

        // Simplified placeholder decode - full implementation requires iq1m_grid lookup
        // Process 8 sub-blocks of 32 elements
        for (size_t ib = 0; ib < 8; ++ib)
        {
            const float scale = d * 2.0f; // Simplified scale extraction
            for (size_t i = 0; i < 32; ++i)
            {
                const size_t idx = ib * 32 + i;
                const uint8_t q = (block.qs[i / 8] >> (i % 8)) & 1;
                output[idx] = scale * (q ? 1.0f : -1.0f);
            }
        }
    }

    IQ1_MTensor::~IQ1_MTensor() {}

    bool IQ1_MTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ1_MTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ1_MBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ1_MTensor::mutable_data()
    {
        throw std::runtime_error("IQ1_MTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> IQ1_MTensor::createRoPE()
    {
        throw std::runtime_error("IQ1_MTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ1_MTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ1_MTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ1_MTensor::createSoftmax()
    {
        throw std::runtime_error("IQ1_MTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ1_MTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ1_MTensor: RMSNorm not supported");
    }

    bool IQ1_MTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ1_MTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
