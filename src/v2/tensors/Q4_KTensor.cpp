/**
 * @file Q4_KTensor.cpp
 * @brief Q4_K quantized tensor implementation (4-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q4_KTensor::Q4_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q4_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q4_KTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void Q4_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(raw_data_.data());
        const Q4_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q4_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q4_KTensor::decodeBlock(const Q4_KBlock &block, float *output)
    {
        // Q4_K: 256 elements in 8 sub-blocks of 32 elements each
        // Each element is 4 bits (2 per byte)
        // Scales: 12 bytes packed as 6-bit scales for 8 sub-blocks
        // d: FP16 main scale, dmin: FP16 minimum offset

        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        // Extract 8 6-bit scales and 8 6-bit mins from 12 bytes
        uint8_t scales[8], mins[8];
        for (size_t i = 0; i < 8; ++i)
        {
            const size_t base = i * 3 / 2; // 3 bytes per 2 scales
            const size_t offset = (i % 2) * 12;

            if (offset == 0)
            {
                // First scale/min in pair
                scales[i] = block.scales[base] & 0x3F;
                mins[i] = block.scales[base + 1] & 0x3F;
            }
            else
            {
                // Second scale/min in pair (spans bytes)
                scales[i] = ((block.scales[base] >> 6) | ((block.scales[base + 1] & 0x0F) << 2)) & 0x3F;
                mins[i] = (block.scales[base + 1] >> 4) & 0x3F;
            }
        }

        // Decode 8 sub-blocks
        for (size_t sub_block = 0; sub_block < 8; ++sub_block)
        {
            const float scale_val = d * static_cast<float>(scales[sub_block]);
            const float min_val = dmin * static_cast<float>(mins[sub_block]);

            // Each sub-block has 32 4-bit elements (16 bytes)
            for (size_t j = 0; j < 32; ++j)
            {
                const size_t idx = sub_block * 32 + j;

                // Get 4-bit value (2 values per byte)
                const size_t qs_idx = sub_block * 16 + j / 2;
                const uint8_t q4 = (j % 2 == 0)
                                       ? (block.qs[qs_idx] & 0x0F) // Lower 4 bits
                                       : (block.qs[qs_idx] >> 4);  // Upper 4 bits

                // Q4_K formula: scale * q - min
                output[idx] = scale_val * static_cast<float>(q4) - min_val;
            }
        }
    }

    Q4_KTensor::~Q4_KTensor() {}

    bool Q4_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_KTensor::mutable_data()
    {
        throw std::runtime_error("Q4_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q4_KTensor::createRoPE()
    {
        throw std::runtime_error("Q4_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q4_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q4_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q4_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q4_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q4_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q4_KTensor: RMSNorm not supported");
    }




    bool Q4_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q4_KTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
