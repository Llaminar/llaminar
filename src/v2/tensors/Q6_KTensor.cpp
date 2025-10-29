/**
 * @file Q6_KTensor.cpp
 * @brief Q6_K quantized tensor implementation (6-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/GemmAutoTuner.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    Q6_KTensor::Q6_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q6_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q6_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q6_KTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q6_KTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void Q6_KTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());
        const Q6_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q6_KTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q6_KTensor::decodeBlock(const Q6_KBlock &block, float *output)
    {
        // Q6_K: 256 elements per super-block, 16 sub-blocks of 16 elements each
        // Each element is 6 bits: 4 bits in ql[], 2 bits in qh[]
        // Decode formula: value = d * scales[sub_block] * reconstruct_6bit(ql[i], qh[i])

        const float d = fp16_to_fp32(block.d);

        // Process 16 sub-blocks
        for (size_t sub_block = 0; sub_block < 16; ++sub_block)
        {
            const float scale = static_cast<float>(block.scales[sub_block]) * d;

            // Each sub-block has 16 elements
            for (size_t j = 0; j < 16; ++j)
            {
                const size_t idx = sub_block * 16 + j;

                // Get 4 lower bits from ql[]
                const uint8_t ql_byte = block.ql[sub_block * 8 + j / 2];
                const uint8_t ql_val = (j % 2 == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);

                // Get 2 higher bits from qh[] (packed 4 per byte)
                const size_t qh_byte_idx = sub_block * 4 + j / 4;
                const size_t qh_bit_shift = (j % 4) * 2;
                const uint8_t qh_val = (block.qh[qh_byte_idx] >> qh_bit_shift) & 0x03;

                // Reconstruct 6-bit value: (qh << 4) | ql
                const int q6 = static_cast<int>((qh_val << 4) | ql_val);

                // Subtract bias and scale (Q6_K uses bias of 32)
                output[idx] = scale * static_cast<float>(q6 - 32);
            }
        }
    }

#if defined(__AVX2__)
    void Q6_KTensor::decodeBlockAVX2(const Q6_KBlock &block, float *output)
    {
        // TODO: Optimize with AVX2 (complex bit unpacking, deferred for initial implementation)
        decodeBlock(block, output);
    }
#endif

    Q6_KTensor::~Q6_KTensor() {}

    bool Q6_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q6_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q6_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q6_KTensor::mutable_data()
    {
        throw std::runtime_error("Q6_KTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q6_KTensor::createRoPE()
    {
        throw std::runtime_error("Q6_KTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q6_KTensor::createSwiGLU()
    {
        throw std::runtime_error("Q6_KTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q6_KTensor::createSoftmax()
    {
        throw std::runtime_error("Q6_KTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q6_KTensor::createRMSNorm()
    {
        throw std::runtime_error("Q6_KTensor: RMSNorm not supported");
    }

    bool Q6_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q6_KTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
