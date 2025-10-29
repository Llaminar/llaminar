/**
 * @file Q5_KTensor.cpp
 * @brief Q5_K quantized tensor implementation (5-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/GemmAutoTuner.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q5_KTensor::Q5_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q5_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q5_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q5_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q5_KTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void Q5_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());
        const Q5_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q5_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q5_KTensor::decodeBlock(const Q5_KBlock &block, float *output)
    {
        // Q5_K: 256 elements, 8 sub-blocks of 32 elements
        // Each element is 5 bits: 4 bits in qs[], 1 bit in qh[]
        // Decode: d * scale * q5 - dmin * min

        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        // Extract 8 6-bit scales from packed bytes (12 bytes → 8×6 bits)
        uint8_t scales[8];
        uint8_t mins[8];
        for (size_t i = 0; i < 8; ++i)
        {
            const size_t base = i * 3 / 2; // 3 bytes per 2 scales
            if (i % 2 == 0)
            {
                scales[i] = block.scales[base] & 0x3F;
                mins[i] = block.scales[base + 1] & 0x3F;
            }
            else
            {
                scales[i] = ((block.scales[base] >> 6) | ((block.scales[base + 1] & 0x0F) << 2));
                mins[i] = (block.scales[base + 1] >> 4);
            }
        }

        // Decode 8 sub-blocks
        for (size_t sub_block = 0; sub_block < 8; ++sub_block)
        {
            const float scale_val = d * static_cast<float>(scales[sub_block]);
            const float min_val = dmin * static_cast<float>(mins[sub_block]);

            // Each sub-block has 32 elements
            for (size_t j = 0; j < 32; ++j)
            {
                const size_t idx = sub_block * 32 + j;

                // Get 4 lower bits from qs[]
                const size_t qs_idx = sub_block * 16 + j / 2;
                const uint8_t qs_val = (j % 2 == 0) ? (block.qs[qs_idx] & 0x0F) : (block.qs[qs_idx] >> 4);

                // Get 1 high bit from qh[] (8 bits per byte)
                const size_t qh_idx = sub_block * 4 + j / 8;
                const size_t qh_shift = j % 8;
                const uint8_t qh_val = (block.qh[qh_idx] >> qh_shift) & 0x01;

                // Reconstruct 5-bit value: (qh << 4) | qs
                const int q5 = static_cast<int>((qh_val << 4) | qs_val);

                output[idx] = scale_val * static_cast<float>(q5) - min_val;
            }
        }
    }

    Q5_KTensor::~Q5_KTensor() {}

    bool Q5_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q5_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q5_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q5_KTensor::mutable_data()
    {
        throw std::runtime_error("Q5_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q5_KTensor::createRoPE()
    {
        throw std::runtime_error("Q5_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q5_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q5_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q5_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q5_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q5_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q5_KTensor: RMSNorm not supported");
    }

    bool Q5_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q5_KTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
