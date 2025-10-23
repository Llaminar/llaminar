/**
 * @file Q8_KTensor.cpp
 * @brief Q8_K quantized tensor implementation (8-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q8_KTensor::Q8_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q8_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q8_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q8_KTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void Q8_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
        const Q8_KBlock *blocks = reinterpret_cast<const Q8_KBlock *>(raw_data_.data());
        const Q8_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q8_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
        const Q8_KBlock *blocks = reinterpret_cast<const Q8_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q8_KTensor::decodeBlock(const Q8_KBlock &block, float *output)
    {
        // Q8_K: 256 int8_t quantized values
        // bsums[16]: Block sums for optimized dot products
        // NOTE: Q8_K has NO scale factor in the block! Scale is stored elsewhere.
        // For now, just convert int8 to float directly (scale will be applied externally)

        for (size_t i = 0; i < Q8_KBlock::BLOCK_SIZE; ++i)
        {
            output[i] = static_cast<float>(block.qs[i]);
        }
    }

    Q8_KTensor::~Q8_KTensor() {}

    bool Q8_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q8_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q8_KBlock *blocks = reinterpret_cast<const Q8_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q8_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q8_KTensor::mutable_data()
    {
        throw std::runtime_error("Q8_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q8_KTensor::createRoPE()
    {
        throw std::runtime_error("Q8_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q8_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q8_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q8_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q8_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q8_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q8_KTensor: RMSNorm not supported");
    }

} // namespace llaminar2
