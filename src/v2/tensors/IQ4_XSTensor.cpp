/**
 * @file IQ4_XSTensor.cpp
 * @brief IQ4_XS quantized tensor implementation (4-bit extra-small IQ, 32-element blocks)
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
    // Note: IQ4_XS uses the same codebook as IQ4_NL (kvalues_iq4nl from IQQuantTables.h)
    // If IQ4_XS requires a different codebook, it should be added to IQQuantTables.h

    IQ4_XSTensor::IQ4_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ4_XSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ4_XSBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ4_XSTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> IQ4_XSTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void IQ4_XSTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());
        const IQ4_XSBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ4_XSTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ4_XSTensor::decodeBlock(const IQ4_XSBlock &block, float *output)
    {
        // IQ4_XS: Uses kvalues_iq4nl codebook lookup (from IQQuantTables.h)
        const float scale = fp16_to_fp32(block.d);

        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t byte = block.qs[i];

            // Extract two 4-bit indices
            const uint8_t idx0 = byte & 0x0F;
            const uint8_t idx1 = byte >> 4;

            // Codebook lookup and scale
            output[i * 2 + 0] = scale * kvalues_iq4nl[idx0];
            output[i * 2 + 1] = scale * kvalues_iq4nl[idx1];
        }
    }

    IQ4_XSTensor::~IQ4_XSTensor() {}

    bool IQ4_XSTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ4_XSTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ4_XSBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ4_XSTensor::mutable_data()
    {
        throw std::runtime_error("IQ4_XSTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> IQ4_XSTensor::createRoPE()
    {
        throw std::runtime_error("IQ4_XSTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ4_XSTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ4_XSTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ4_XSTensor::createSoftmax()
    {
        throw std::runtime_error("IQ4_XSTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ4_XSTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ4_XSTensor: RMSNorm not supported");
    }




    bool IQ4_XSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ4_XSTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
