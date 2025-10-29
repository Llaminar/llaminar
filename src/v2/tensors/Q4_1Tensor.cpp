/**
 * @file Q4_1Tensor.cpp
 * @brief Q4_1 quantized tensor implementation (4-bit per weight with min, ~7.1x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/GemmAutoTuner.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    Q4_1Tensor::Q4_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q4_1Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_1Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_1Tensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q4_1Tensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void Q4_1Tensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());
        const Q4_1Block &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q4_1Tensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q4_1Tensor::decodeBlock(const Q4_1Block &block, float *output)
    {
#if defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback: decode formula is scale * nibble + min
        const float scale = fp16_to_fp32(block.d);
        const float min = fp16_to_fp32(block.m);

        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t byte = block.qs[i];

            // Extract two 4-bit nibbles (no bias subtraction for Q4_1)
            const float v0 = static_cast<float>(byte & 0x0F);
            const float v1 = static_cast<float>(byte >> 4);

            output[i * 2 + 0] = scale * v0 + min;
            output[i * 2 + 1] = scale * v1 + min;
        }
#endif
    }

#if defined(__AVX2__)
    void Q4_1Tensor::decodeBlockAVX2(const Q4_1Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const float min = fp16_to_fp32(block.m);
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256 vmin = _mm256_set1_ps(min);

        // Load 16 bytes (contains 32 4-bit values)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
        __m256i v4bit_256 = _mm256_set_m128i(v4bit, v4bit);

        // Extract low and high nibbles
        __m256i vlow = _mm256_and_si256(v4bit_256, _mm256_set1_epi8(0x0F));
        __m256i vhigh = _mm256_and_si256(_mm256_srli_epi16(v4bit_256, 4), _mm256_set1_epi8(0x0F));

        // Process in chunks of 8 elements
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            __m128i vlow_chunk = (chunk == 0) ? _mm256_castsi256_si128(vlow) : _mm256_extracti128_si256(vlow, 1);
            __m128i vhigh_chunk = (chunk == 0) ? _mm256_castsi256_si128(vhigh) : _mm256_extracti128_si256(vhigh, 1);

            // Convert first 8 low nibbles to float
            __m256i vi32_low = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(&vlow_chunk)));
            __m256 vf_low = _mm256_cvtepi32_ps(vi32_low);
            vf_low = _mm256_fmadd_ps(vf_low, vscale, vmin); // scale * value + min
            _mm256_storeu_ps(&output[chunk * 16], vf_low);

            // Convert first 8 high nibbles to float
            __m256i vi32_high = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(&vhigh_chunk)));
            __m256 vf_high = _mm256_cvtepi32_ps(vi32_high);
            vf_high = _mm256_fmadd_ps(vf_high, vscale, vmin);
            _mm256_storeu_ps(&output[chunk * 16 + 8], vf_high);
        }
    }
#endif

    Q4_1Tensor::~Q4_1Tensor() {}

    bool Q4_1Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_1Tensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_1Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_1Tensor::mutable_data()
    {
        throw std::runtime_error("Q4_1Tensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q4_1Tensor::createRoPE()
    {
        throw std::runtime_error("Q4_1Tensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q4_1Tensor::createSwiGLU()
    {
        throw std::runtime_error("Q4_1Tensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q4_1Tensor::createSoftmax()
    {
        throw std::runtime_error("Q4_1Tensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q4_1Tensor::createRMSNorm()
    {
        throw std::runtime_error("Q4_1Tensor: RMSNorm not supported");
    }

    bool Q4_1Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q4_1Tensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
