/**
 * @file Q2_KTensor.cpp
 * @brief Q2_K quantized tensor implementation (2-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q2_KTensor::Q2_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q2_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q2_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q2_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q2_KTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void Q2_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
        const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());
        const Q2_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q2_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
        const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q2_KTensor::decodeBlock(const Q2_KBlock &block, float *output)
    {
        // Q2_K: 256 elements, 16 sub-blocks of 16 elements
        // scales[] contains packed scales and mins (4 bits each)
        // qs[] contains 2-bit quantized values (4 per byte)
        // Decode: d * scale * q - dmin * min

        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        // Extract scales and mins from packed bytes
        uint8_t scales[16];
        uint8_t mins[16];
        for (size_t i = 0; i < 16; ++i)
        {
            const size_t byte_idx = i / 2;
            if (i % 2 == 0)
            {
                scales[i] = block.scales[byte_idx] & 0x0F;
                mins[i] = block.scales[byte_idx + 8] & 0x0F;
            }
            else
            {
                scales[i] = block.scales[byte_idx] >> 4;
                mins[i] = block.scales[byte_idx + 8] >> 4;
            }
        }

        // Decode 16 sub-blocks
        for (size_t sub_block = 0; sub_block < 16; ++sub_block)
        {
            const float scale_val = d * static_cast<float>(scales[sub_block]);
            const float min_val = dmin * static_cast<float>(mins[sub_block]);

            // Each sub-block has 16 elements (4 bytes, 4 values per byte)
            for (size_t j = 0; j < 16; ++j)
            {
                const size_t idx = sub_block * 16 + j;
                const size_t byte_idx = sub_block * 4 + j / 4;
                const size_t bit_shift = (j % 4) * 2;

                const uint8_t q2 = (block.qs[byte_idx] >> bit_shift) & 0x03;
                output[idx] = scale_val * static_cast<float>(q2) - min_val;
            }
        }
    }

    Q2_KTensor::~Q2_KTensor() {}

    bool Q2_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q2_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q2_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q2_KTensor::mutable_data()
    {
        throw std::runtime_error("Q2_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q2_KTensor::createRoPE()
    {
        throw std::runtime_error("Q2_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q2_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q2_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q2_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q2_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q2_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q2_KTensor: RMSNorm not supported");
    }




    bool Q2_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q2_KTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
