/**
 * @file Q3_KTensor.cpp
 * @brief Q3_K quantized tensor implementation (3-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q3_KTensor::Q3_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q3_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q3_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q3_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q3_KTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void Q3_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());
        const Q3_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q3_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q3_KTensor::decodeBlock(const Q3_KBlock &block, float *output)
    {
        // Q3_K: 256 elements, 16 sub-blocks of 16 elements
        // Each element is 3 bits: 2 bits in qs[], 1 bit in hmask[]
        // Decode: d * scale * reconstruct_3bit(qs[i], hmask[i])

        const float d = fp16_to_fp32(block.d);

        // Extract 16 6-bit scales from packed bytes (12 bytes → 16×6 bits)
        uint8_t scales[16];
        for (size_t i = 0; i < 16; ++i)
        {
            const size_t base = i * 3 / 4; // 3 bytes per 4 scales
            const size_t offset = (i % 4) * 6;

            if (offset <= 2)
            {
                scales[i] = (block.scales[base] >> offset) & 0x3F;
            }
            else
            {
                // Scale spans two bytes
                const uint8_t low_bits = block.scales[base] >> offset;
                const uint8_t high_bits = block.scales[base + 1] << (8 - offset);
                scales[i] = (low_bits | high_bits) & 0x3F;
            }
        }

        // Decode 16 sub-blocks
        for (size_t sub_block = 0; sub_block < 16; ++sub_block)
        {
            const float scale_val = d * static_cast<float>(scales[sub_block]);

            // Each sub-block has 16 elements
            for (size_t j = 0; j < 16; ++j)
            {
                const size_t idx = sub_block * 16 + j;

                // Get 2 lower bits from qs[] (4 values per byte)
                const size_t qs_idx = sub_block * 4 + j / 4;
                const size_t qs_shift = (j % 4) * 2;
                const uint8_t qs_val = (block.qs[qs_idx] >> qs_shift) & 0x03;

                // Get 1 high bit from hmask[] (8 bits per byte)
                const size_t hm_idx = sub_block * 2 + j / 8;
                const size_t hm_shift = j % 8;
                const uint8_t hm_val = (block.hmask[hm_idx] >> hm_shift) & 0x01;

                // Reconstruct 3-bit value: (hm << 2) | qs
                const int q3 = static_cast<int>((hm_val << 2) | qs_val);

                // Q3_K uses bias of 4
                output[idx] = scale_val * static_cast<float>(q3 - 4);
            }
        }
    }

    Q3_KTensor::~Q3_KTensor() {}

    bool Q3_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q3_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q3_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q3_KTensor::mutable_data()
    {
        throw std::runtime_error("Q3_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q3_KTensor::createRoPE()
    {
        throw std::runtime_error("Q3_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q3_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q3_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q3_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q3_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q3_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q3_KTensor: RMSNorm not supported");
    }




    bool Q3_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q3_KTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
